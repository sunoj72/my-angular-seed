(function(){
  'use strict';

  angular.module('ews.ui.mxgraph')
    .directive('ewsMxgraph', MxGraphDirective);

  MxGraphDirective.$inject = ['$log'];

  function MxGraphDirective ($log) {
    return {
      restrict: 'E',
      name: 'ews-mxgraph',
      scope: true,
      template: '<div flex class="mx-graph" style="left:0;top:0;position:relative;"></div>',
      controller: 'MxGraphController',
      controllerAs: 'mxgraph',
      // bindToController: true,
      link: linker
    };


    /// internal functions

    function linker (scope, elements, attrs, controller) {
      var self = this;

      controller.graphUI = controller.graphUI || {};
      controller.graphUI.readonly = true,
      controller.graphUI.monitor = false;
      controller.graphUI.model = controller.graphUI.model || {};
      controller.graphUI.container = elements[0].children[0];

      // Attributes : ews-mode
      if (angular.isUndefined(attrs.ewsMode)) {
        attrs.ewsMode = 'viewer';
      }

      if (attrs.ewsMode === 'editor') {
        controller.graphUI.readonly = false;
      } else {
        controller.graphUI.readonly = true;

        if (attrs.ewsMode === 'monitor') {
          controller.graphUI.monitor = true;
        }
      }

      if (!mxClient.isBrowserSupported())
      {
        mxUtils.error('Browser is not supported!', 200, false);
        return null;
      }

      controller.initGraphEditor();

      scope.$on('destroy', function() {
        $log.info('destory ews-mxgraph');
      });
    }
  }

})();

